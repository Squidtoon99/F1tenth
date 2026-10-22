#!/usr/bin/env bash
# SSH-install the IRL Gigaflow car image, Sophy GT car images, and the
# standalone trainer. Wraps load_to_jetson.sh for docker tarballs; copies
# trainer Apptainer (.sif) or a trainer docker save to the remote host.
#
# Usage:
#   deploy/scripts/install_irl_images.sh user@car01 car01 \
#     --gigaflow deploy/snapshots/f1tenth-gigaflow-<sha>.tar.gz \
#     --gigaflow-ckpt gigaflow_policy.pt \
#     --sensor-policy deploy/snapshots/f1tenth-sensor-policy-<sha>.tar.gz \
#     --racing deploy/snapshots/f1tenth-racing-<sha>.tar.gz \
#     --training training.sif
#
#   deploy/scripts/install_irl_images.sh user@gpu-box - \
#     --training training.sif
#
# Flags:
#   --gigaflow TAR            f1tenth-gigaflow docker save (.tar.gz)
#   --gigaflow-ckpt PT        baked actor (actor_005700 until fine-tune lands)
#   --sensor-policy TAR       f1tenth-sensor-policy docker save
#   --sensor-policy-ckpt PT   optional e2e_policy.pt
#   --racing TAR              f1tenth-racing docker save
#   --training PATH           training.sif or a trainer docker .tar/.tar.gz
#   --training-sif SIF        alias for Apptainer from deploy/apptainer/training.def
#   --training-docker TAR     alias for a trainer docker save
#   --sophy-car TAR           alias for --sensor-policy
#   --sophy-ckpt PT           alias for --sensor-policy-ckpt
#   --images-dir DIR          remote trainer dest (default /opt/f1tenth/images)
#
# Does not bake weights into images, does not start training, does not overwrite
# /opt/f1tenth/policies/policy.pt or another stack's moving tag.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

usage() {
  sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
}

HOST=""
CAR=""
GIGAFLOW_TAR=""
GIGAFLOW_CKPT=""
SOPHY_CAR_TAR=""
SOPHY_CKPT=""
RACING_TAR=""
TRAINING_SIF=""
TRAINING_DOCKER=""
IMAGES_DIR="/opt/f1tenth/images"

_assign_training() {
  local path="$1"
  case "${path}" in
    *.sif) TRAINING_SIF="${path}" ;;
    *.tar|*.tar.gz|*.tgz) TRAINING_DOCKER="${path}" ;;
    *)
      echo "--training expects a .sif or docker .tar/.tar.gz (got ${path})" >&2
      exit 1
      ;;
  esac
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

HOST="${1:?usage: install_irl_images.sh <user@host> <carNN|-> [flags]}"
shift
if [ $# -gt 0 ] && [[ "${1}" != --* ]]; then
  CAR="${1}"
  shift
fi

while [ $# -gt 0 ]; do
  case "${1}" in
    --gigaflow) GIGAFLOW_TAR="${2:?--gigaflow needs a .tar.gz}"; shift 2 ;;
    --gigaflow-ckpt) GIGAFLOW_CKPT="${2:?--gigaflow-ckpt needs a .pt}"; shift 2 ;;
    --sensor-policy|--sophy-car) SOPHY_CAR_TAR="${2:?--sensor-policy needs a .tar.gz}"; shift 2 ;;
    --sensor-policy-ckpt|--sophy-ckpt) SOPHY_CKPT="${2:?--sensor-policy-ckpt needs a .pt}"; shift 2 ;;
    --racing) RACING_TAR="${2:?--racing needs a .tar.gz}"; shift 2 ;;
    --training) _assign_training "${2:?--training needs a path}"; shift 2 ;;
    --training-sif) TRAINING_SIF="${2:?--training-sif needs a .sif}"; shift 2 ;;
    --training-docker) TRAINING_DOCKER="${2:?--training-docker needs a tar}"; shift 2 ;;
    --images-dir) IMAGES_DIR="${2:?--images-dir needs a path}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "unknown flag: ${1}" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "${GIGAFLOW_TAR}" ] && [ -z "${TRAINING_SIF}" ] && [ -z "${TRAINING_DOCKER}" ] \
    && [ -z "${SOPHY_CAR_TAR}" ] && [ -z "${RACING_TAR}" ]; then
  echo "need at least --gigaflow, --sensor-policy, --racing, or --training" >&2
  exit 1
fi
if [ -n "${GIGAFLOW_TAR}" ] || [ -n "${SOPHY_CAR_TAR}" ] || [ -n "${RACING_TAR}" ]; then
  if [ -z "${CAR}" ] || [ "${CAR}" = "-" ]; then
    echo "car id required when installing a car image (got '${CAR:-}')" >&2
    exit 1
  fi
fi

_need_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "${path}" ]; then
    echo "${label} not found: ${path}" >&2
    exit 1
  fi
}

_sha() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

LOAD_TO_JETSON="deploy/scripts/load_to_jetson.sh"
INVENTORY_REMOTE="/opt/f1tenth/rollback/irl/inventory.txt"
NOTES="irl_install"

if [ -n "${GIGAFLOW_TAR}" ]; then
  _need_file "${GIGAFLOW_TAR}" "--gigaflow"
  NOTES="${NOTES} gigaflow=$(_sha "${GIGAFLOW_TAR}")"
  echo "==> Gigaflow car image -> ${HOST} (${CAR})"
  if [ -n "${GIGAFLOW_CKPT}" ]; then
    _need_file "${GIGAFLOW_CKPT}" "--gigaflow-ckpt"
    NOTES="${NOTES} gigaflow_ckpt=$(_sha "${GIGAFLOW_CKPT}")"
    TARGET=gigaflow CHECKPOINT="${GIGAFLOW_CKPT}" \
      "${LOAD_TO_JETSON}" "${HOST}" "${GIGAFLOW_TAR}" "${CAR}"
  else
    TARGET=gigaflow "${LOAD_TO_JETSON}" "${HOST}" "${GIGAFLOW_TAR}" "${CAR}"
  fi
fi

if [ -n "${SOPHY_CAR_TAR}" ]; then
  _need_file "${SOPHY_CAR_TAR}" "--sensor-policy"
  NOTES="${NOTES} sensor_policy=$(_sha "${SOPHY_CAR_TAR}")"
  echo "==> Sensor-policy (Sophy car) image -> ${HOST} (${CAR})"
  if [ -n "${SOPHY_CKPT}" ]; then
    _need_file "${SOPHY_CKPT}" "--sensor-policy-ckpt"
    NOTES="${NOTES} sensor_policy_ckpt=$(_sha "${SOPHY_CKPT}")"
    TARGET=sensor_policy CHECKPOINT="${SOPHY_CKPT}" \
      "${LOAD_TO_JETSON}" "${HOST}" "${SOPHY_CAR_TAR}" "${CAR}"
  else
    TARGET=sensor_policy "${LOAD_TO_JETSON}" "${HOST}" "${SOPHY_CAR_TAR}" "${CAR}"
  fi
fi

if [ -n "${RACING_TAR}" ]; then
  _need_file "${RACING_TAR}" "--racing"
  NOTES="${NOTES} racing=$(_sha "${RACING_TAR}")"
  echo "==> Racing image -> ${HOST} (${CAR})"
  TARGET=racing "${LOAD_TO_JETSON}" "${HOST}" "${RACING_TAR}" "${CAR}"
fi

if [ -n "${TRAINING_SIF}" ] || [ -n "${TRAINING_DOCKER}" ]; then
  echo "==> Trainer images -> ${HOST}:${IMAGES_DIR}"
  ssh "${HOST}" "mkdir -p '${IMAGES_DIR}' /opt/f1tenth/rollback/irl"
fi

if [ -n "${TRAINING_SIF}" ]; then
  _need_file "${TRAINING_SIF}" "--training-sif"
  NOTES="${NOTES} training_sif=$(_sha "${TRAINING_SIF}")"
  echo "==> Copying Apptainer training.sif (standalone_trainer / Sophy GT)"
  scp "${TRAINING_SIF}" "${HOST}:${IMAGES_DIR}/training.sif"
  ssh "${HOST}" bash -s -- "${IMAGES_DIR}" <<'EOF'
set -euo pipefail
IMAGES_DIR="$1"
cat > "${IMAGES_DIR}/run_training.sh" <<RUN
#!/usr/bin/env bash
set -euo pipefail
SIF="\${TRAINING_SIF:-${IMAGES_DIR}/training.sif}"
if ! command -v apptainer >/dev/null 2>&1; then
  echo "apptainer not found on PATH; load \${SIF} on a host that has it." >&2
  exit 1
fi
exec apptainer run --nv "\${SIF}" "\$@"
RUN
chmod +x "${IMAGES_DIR}/run_training.sh"
ls -la "${IMAGES_DIR}/training.sif" "${IMAGES_DIR}/run_training.sh"
EOF
fi

if [ -n "${TRAINING_DOCKER}" ]; then
  _need_file "${TRAINING_DOCKER}" "--training-docker"
  NOTES="${NOTES} training_docker=$(_sha "${TRAINING_DOCKER}")"
  REMOTE_TAR="/tmp/$(basename "${TRAINING_DOCKER}")"
  echo "==> Loading trainer docker save on ${HOST}"
  scp "${TRAINING_DOCKER}" "${HOST}:${REMOTE_TAR}"
  ssh "${HOST}" bash -s -- "${REMOTE_TAR}" <<'EOF'
set -euo pipefail
TAR="$1"
if ! command -v docker >/dev/null 2>&1; then
  echo "docker not usable on remote; cannot load ${TAR}" >&2
  exit 1
fi
if [[ "${TAR}" == *.gz ]]; then
  gunzip -c "${TAR}" | docker load
else
  docker load -i "${TAR}"
fi
rm -f "${TAR}"
docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
EOF
fi

GITSHA="$(git rev-parse --short HEAD)"
CAR_NOTE="${CAR:--}"
ssh "${HOST}" bash -s -- "${INVENTORY_REMOTE}" "${GITSHA}" "${NOTES}" <<'EOF'
set -euo pipefail
INV="$1"
GITSHA="$2"
NOTES="$3"
mkdir -p "$(dirname "${INV}")"
{
  echo "git_sha=${GITSHA}"
  echo "installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "notes=${NOTES}"
  echo "-- docker images --"
  docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' 2>/dev/null \
    | grep -E 'f1tenth-(gigaflow|sensor-policy|racing|dev)|training' || true
  echo "-- trainer files --"
  ls -la /opt/f1tenth/images 2>/dev/null || true
} > "${INV}"
cat "${INV}"
EOF

MANIFEST="deploy/releases/manifest.csv"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "${GITSHA},irl-install,${CAR_NOTE},${STAMP},${NOTES}" >> "${MANIFEST}"
ssh "${HOST}" bash -s -- "${GITSHA}" "${CAR_NOTE}" "${STAMP}" "${NOTES}" <<'EOF'
set -euo pipefail
mkdir -p /opt/f1tenth
printf '%s\n' "$1,irl-install,$2,$3,$4" >> /opt/f1tenth/manifest.csv
EOF

cat <<EOF

==> IRL image install on ${HOST} complete
    inventory: ${INVENTORY_REMOTE}
    manifest:  ${MANIFEST}

Car Gigaflow (if loaded):
  ssh ${HOST} /opt/f1tenth/rollback/gigaflow/run.sh

Sophy GT / standalone trainer (if copied):
  ssh ${HOST} ${IMAGES_DIR}/run_training.sh --help
EOF
