#!/usr/bin/env bash
# One-command install of the DarkToaster IRL deploy bundle onto the F1TENTH car.
#
# Usage:
#   ./install_to_shereef_f1tenth.sh [--dry-run] [--bundle-dir DIR]
#       [--build-on-car|--load-only] [user@host] [car_id]
#
# Defaults: user@host=shereef@f1tenth  car_id=car01
#
# Env:
#   SKIP_TRAINER=1  do not rsync/load the amd64 courtyard-trainer tarball
#   SKIP_AMD64=1    do not rsync linux/amd64 CPU car tarballs (Jetson path)
#
# Transfers image tarballs, checkpoints, params.yaml, and car-build-context in
# one rsync, then docker-loads images (or builds them natively) and writes
# /opt/f1tenth/run/*.sh wrappers on the car.
# On the target, `uname -m` selects linux/arm64 (`*-arm64.tar.gz`) vs
# linux/amd64 tarballs. Default is auto-fallback: docker load the CUDA arm64
# tarballs, and if a tarball is missing or load fails, run
# car-build-context/build_on_car.sh on the Jetson.
# --build-on-car forces the native NGC iGPU rebuild. --load-only never builds.
#
# Does NOT SSH when --dry-run is set. Does not delete remote files.
set -euo pipefail

usage() {
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
BUNDLE_DIR=""
HOST=""
CAR=""
BUILD_ON_CAR="auto"

while [ $# -gt 0 ]; do
  case "${1}" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --bundle-dir)
      BUNDLE_DIR="${2:?--bundle-dir needs a path}"
      shift 2
      ;;
    --build-on-car)
      BUILD_ON_CAR="1"
      shift
      ;;
    --load-only)
      BUILD_ON_CAR="0"
      shift
      ;;
    --*)
      echo "unknown flag: ${1}" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [ -z "${HOST}" ]; then
        HOST="${1}"
      elif [ -z "${CAR}" ]; then
        CAR="${1}"
      else
        echo "unexpected extra argument: ${1}" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [ -z "${BUNDLE_DIR}" ]; then
  if [ -d "${SCRIPT_DIR}/policies" ] && [ -d "${SCRIPT_DIR}/car-build-context" ]; then
    BUNDLE_DIR="${SCRIPT_DIR}"
  elif [ -d "${SCRIPT_DIR}/images" ] && [ -d "${SCRIPT_DIR}/policies" ]; then
    BUNDLE_DIR="${SCRIPT_DIR}"
  elif [ -d /home/ubuntu/f1tenth-irl-deploy-bundle/images ]; then
    BUNDLE_DIR=/home/ubuntu/f1tenth-irl-deploy-bundle
  else
    BUNDLE_DIR="${SCRIPT_DIR}"
  fi
fi
BUNDLE_DIR="$(cd "${BUNDLE_DIR}" && pwd)"
HOST="${HOST:-shereef@f1tenth}"
CAR="${CAR:-car01}"
SKIP_TRAINER="${SKIP_TRAINER:-0}"
SKIP_AMD64="${SKIP_AMD64:-0}"

REMOTE_ROOT="/opt/f1tenth"
REMOTE_BUNDLE="${REMOTE_ROOT}/deploy-bundle"
REMOTE_POLICIES="${REMOTE_ROOT}/policies"
REMOTE_CONFIG="${REMOTE_ROOT}/config"
REMOTE_RUN="${REMOTE_ROOT}/run"
REMOTE_IMAGES="${REMOTE_ROOT}/images"

IMAGES_DIR="${BUNDLE_DIR}/images"
POLICIES_DIR="${BUNDLE_DIR}/policies"
CONFIG_DIR="${BUNDLE_DIR}/config"
if [ ! -d "${POLICIES_DIR}" ] && [ -f "${BUNDLE_DIR}/courtyard_sleepy_s2_dagger_7lap.pt" ]; then
  POLICIES_DIR="${BUNDLE_DIR}"
fi
if [ ! -d "${CONFIG_DIR}" ] && [ -f "${BUNDLE_DIR}/params.yaml" ]; then
  CONFIG_DIR="${BUNDLE_DIR}"
fi

need_file() {
  local path="$1"
  if [ ! -f "${path}" ]; then
    echo "missing required file: ${path}" >&2
    return 1
  fi
}

echo "==> IRL deploy bundle"
echo "    bundle:  ${BUNDLE_DIR}"
echo "    target:  ${HOST}  car=${CAR}"
echo "    dry-run: ${DRY_RUN}"
echo "    build:   ${BUILD_ON_CAR}  (auto=load CUDA arm64 tarballs, native-build fallback)"
echo "    skip:    trainer=${SKIP_TRAINER} amd64=${SKIP_AMD64}"
echo

MISSING=0
need_file "${CONFIG_DIR}/params.yaml" || MISSING=1
if [ ! -d "${POLICIES_DIR}" ]; then
  echo "missing policies dir: ${POLICIES_DIR}" >&2
  MISSING=1
fi
if [ ! -d "${IMAGES_DIR}" ] && [ "${BUILD_ON_CAR}" != "1" ]; then
  echo "missing images dir: ${IMAGES_DIR}" >&2
  MISSING=1
fi

GIGAFLOW_ALL="${IMAGES_DIR}/f1tenth-gigaflow-all-models.tar.gz"
GIGAFLOW_ONE="${IMAGES_DIR}/f1tenth-gigaflow.tar.gz"
GIGAFLOW_ARM64="${IMAGES_DIR}/f1tenth-gigaflow-arm64.tar.gz"
SENSOR_TAR="${IMAGES_DIR}/f1tenth-sensor-policy.tar.gz"
SENSOR_ALL="${IMAGES_DIR}/f1tenth-sensor-policy-all-models.tar.gz"
SENSOR_ARM64="${IMAGES_DIR}/f1tenth-sensor-policy-arm64.tar.gz"
TRAINER_TAR="${IMAGES_DIR}/f1tenth-courtyard-trainer.tar.gz"
RACING_TAR="${IMAGES_DIR}/f1tenth-racing.tar.gz"
RACING_ARM64="${IMAGES_DIR}/f1tenth-racing-arm64.tar.gz"

# Jetson Orin (`shereef@f1tenth`) is aarch64 — never load amd64 CPU images on the car.
CAR_CTX="${BUNDLE_DIR}/car-build-context"
if [ ! -x "${CAR_CTX}/build_on_car.sh" ]; then
  echo "missing native on-car build context: ${CAR_CTX}/build_on_car.sh" >&2
  MISSING=1
fi
if [ "${BUILD_ON_CAR}" != "1" ]; then
  need_file "${GIGAFLOW_ARM64}" || MISSING=1
  need_file "${SENSOR_ARM64}" || MISSING=1
fi
if [ ! -f "${RACING_ARM64}" ] && [ "${BUILD_ON_CAR}" = "0" ]; then
  echo "missing f1tenth-racing-arm64.tar.gz and --load-only was set" >&2
  MISSING=1
fi
if [ "${SKIP_AMD64}" != "1" ] && [ "${BUILD_ON_CAR}" != "1" ]; then
  if [ ! -f "${GIGAFLOW_ALL}" ] && [ ! -f "${GIGAFLOW_ONE}" ]; then
    echo "missing amd64 gigaflow tarball (want f1tenth-gigaflow-all-models.tar.gz or f1tenth-gigaflow.tar.gz)" >&2
    MISSING=1
  fi
  if [ ! -f "${SENSOR_ALL}" ] && [ ! -f "${SENSOR_TAR}" ]; then
    echo "missing amd64 sensor-policy tarball" >&2
    MISSING=1
  fi
  if [ ! -f "${RACING_TAR}" ]; then
    echo "missing racing tarball" >&2
    MISSING=1
  fi
fi
if [ "${SKIP_TRAINER}" != "1" ] && [ ! -f "${TRAINER_TAR}" ]; then
  echo "missing courtyard trainer tarball" >&2
  MISSING=1
fi

POLICIES=(
  courtyard_sleepy_s2_dagger_7lap.pt
  courtyard_pr104_hybrid_bc_6lap.pt
  irl_awac_kl_kl2x_winner.pt
  actor_005700_irl_baked.pt
  gigaflow_folded_005700.pt
  gigaflow_folded_sophy_8m.pt
  policy_875520000.pt
  policy_768188416.pt
)
for pt in "${POLICIES[@]}"; do
  need_file "${POLICIES_DIR}/${pt}" || MISSING=1
done

if [ "${MISSING}" -ne 0 ] && [ "${DRY_RUN}" -eq 0 ]; then
  echo "bundle incomplete; refusing to SSH" >&2
  exit 1
fi
if [ "${MISSING}" -ne 0 ]; then
  echo "WARNING: bundle incomplete; dry-run continues so the plan can be reviewed" >&2
fi

echo "==> Local artifacts"
ls -lh "${IMAGES_DIR}"/*.tar.gz 2>/dev/null || echo "    (no image tarballs yet)"
ls -lh "${POLICIES_DIR}"/*.pt
echo

run() {
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

remote() {
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf 'DRY-RUN: ssh %q bash -s\n' "${HOST}"
    printf '%s\n' "$1" | sed 's/^/        /'
    return 0
  fi
  ssh "${HOST}" bash -s <<EOF
$1
EOF
}

echo "==> 1. mkdir remote layout + rsync bundle"
run ssh "${HOST}" "mkdir -p '${REMOTE_BUNDLE}' '${REMOTE_POLICIES}' '${REMOTE_CONFIG}' '${REMOTE_RUN}' '${REMOTE_IMAGES}'"
RSYNC_IMAGE_EXCLUDES=()
if [ "${SKIP_TRAINER}" = "1" ]; then
  RSYNC_IMAGE_EXCLUDES+=(--exclude 'f1tenth-courtyard-trainer.tar.gz')
fi
if [ "${SKIP_AMD64}" = "1" ]; then
  RSYNC_IMAGE_EXCLUDES+=(
    --exclude 'f1tenth-gigaflow.tar.gz'
    --exclude 'f1tenth-gigaflow-all-models.tar.gz'
    --exclude 'f1tenth-sensor-policy.tar.gz'
    --exclude 'f1tenth-sensor-policy-all-models.tar.gz'
    --exclude 'f1tenth-racing.tar.gz'
    --exclude 'f1tenth-courtyard-trainer.tar.gz'
  )
fi
if [ -d "${IMAGES_DIR}" ]; then
  run rsync -aP --info=progress2 \
    "${RSYNC_IMAGE_EXCLUDES[@]}" \
    "${BUNDLE_DIR}/images/" "${HOST}:${REMOTE_BUNDLE}/images/"
else
  echo "    (no images/ dir; --build-on-car will native-build on the Jetson)"
fi
run rsync -aP --info=progress2 \
  "${BUNDLE_DIR}/policies/" "${HOST}:${REMOTE_BUNDLE}/policies/"
run rsync -aP --info=progress2 \
  "${BUNDLE_DIR}/config/" "${HOST}:${REMOTE_BUNDLE}/config/"
if [ -f "${BUNDLE_DIR}/SHA256SUMS" ]; then
  run rsync -aP "${BUNDLE_DIR}/SHA256SUMS" "${HOST}:${REMOTE_BUNDLE}/SHA256SUMS"
fi
if [ -d "${BUNDLE_DIR}/car-build-context" ]; then
  run rsync -aP --info=progress2 \
    "${BUNDLE_DIR}/car-build-context/" "${HOST}:${REMOTE_BUNDLE}/car-build-context/"
fi

echo "==> 2. stage policies + params.yaml (does not delete existing files)"
remote "$(cat <<'EOS'
set -euo pipefail
REMOTE_ROOT=/opt/f1tenth
BUNDLE=${REMOTE_ROOT}/deploy-bundle
mkdir -p ${REMOTE_ROOT}/policies ${REMOTE_ROOT}/config ${REMOTE_ROOT}/run ${REMOTE_ROOT}/images
cp -an ${BUNDLE}/policies/. ${REMOTE_ROOT}/policies/
cp -n ${BUNDLE}/config/params.yaml ${REMOTE_ROOT}/config/params.yaml 2>/dev/null \
  || cp ${BUNDLE}/config/params.yaml ${REMOTE_ROOT}/config/params.yaml
if [ -d ${BUNDLE}/config/maps ]; then
  mkdir -p ${REMOTE_ROOT}/config/maps
  cp -an ${BUNDLE}/config/maps/. ${REMOTE_ROOT}/config/maps/ || true
fi
if [ -f ${BUNDLE}/config/car.yaml ] && [ ! -f ${REMOTE_ROOT}/config/car.yaml ]; then
  cp ${BUNDLE}/config/car.yaml ${REMOTE_ROOT}/config/car.yaml
fi
ln -sfn courtyard_sleepy_s2_dagger_7lap.pt ${REMOTE_ROOT}/policies/gigaflow_policy.pt
ln -sfn gigaflow_folded_sophy_8m.pt ${REMOTE_ROOT}/policies/e2e_policy.pt
if [ -f ${BUNDLE}/SHA256SUMS ]; then
  (cd ${REMOTE_ROOT}/policies && sha256sum -c ${BUNDLE}/SHA256SUMS --ignore-missing) || true
fi
ls -lh ${REMOTE_ROOT}/policies ${REMOTE_ROOT}/config/params.yaml
EOS
)"

echo "==> 3. docker load CUDA arm64 tarballs (or native --build-on-car fallback)"
remote "$(cat <<EOS
set -euo pipefail
BUILD_ON_CAR='${BUILD_ON_CAR}'
SKIP_TRAINER='${SKIP_TRAINER}'
if ! command -v docker >/dev/null 2>&1; then
  echo "docker not usable on remote" >&2
  exit 1
fi
IMAGES=/opt/f1tenth/deploy-bundle/images
CTX=/opt/f1tenth/deploy-bundle/car-build-context
ARCH=\$(uname -m)
echo "==> target uname -m: \${ARCH}  BUILD_ON_CAR=\${BUILD_ON_CAR}"
load_tar() {
  local tar="\$1"
  echo "==> docker load \${tar}"
  if [ ! -f "\${tar}" ]; then
    echo "tarball missing on car: \${tar}" >&2
    return 1
  fi
  gunzip -c "\${tar}" | docker load
}
retag() {
  local src="\$1" dest="\$2"
  if docker image inspect "\${src}" >/dev/null 2>&1; then
    docker tag "\${src}" "\${dest}"
    echo "tagged \${src} -> \${dest}"
  fi
}
native_build() {
  local target="\$1"
  if [ ! -x "\${CTX}/build_on_car.sh" ]; then
    echo "car-build-context missing; cannot native-build \${target}" >&2
    return 1
  fi
  echo "==> native on-car build TARGET=\${target}"
  "\${CTX}/build_on_car.sh" "\${target}"
}
load_or_build() {
  local tar="\$1"
  local target="\$2"
  if [ "\${BUILD_ON_CAR}" = "1" ]; then
    native_build "\${target}"
    return
  fi
  if load_tar "\${tar}"; then
    return 0
  fi
  if [ "\${BUILD_ON_CAR}" = "0" ]; then
    echo "load failed and --load-only is set: \${tar}" >&2
    return 1
  fi
  echo "auto-fallback: native build \${target} (tarball missing or docker load failed)"
  native_build "\${target}"
}
case "\${ARCH}" in
  aarch64|arm64)
    load_or_build "\${IMAGES}/f1tenth-gigaflow-arm64.tar.gz" gigaflow
    load_or_build "\${IMAGES}/f1tenth-sensor-policy-arm64.tar.gz" sensor_policy
    load_or_build "\${IMAGES}/f1tenth-racing-arm64.tar.gz" racing
    echo "note: courtyard trainer is linux/amd64 CUDA-host; not loaded on Jetson"
    retag f1tenth-gigaflow:gigaflow-arm64 f1tenth-gigaflow:gigaflow
    retag f1tenth-gigaflow:courtyard-sleepy-7lap-arm64 f1tenth-gigaflow:courtyard-sleepy-7lap
    retag f1tenth-gigaflow:courtyard-pr104-6lap-arm64 f1tenth-gigaflow:courtyard-pr104-6lap
    retag f1tenth-gigaflow:irl-awac-kl-arm64 f1tenth-gigaflow:irl-awac-kl
    retag f1tenth-gigaflow:base-5700-arm64 f1tenth-gigaflow:base-5700
    retag f1tenth-sensor-policy:sensor-policy-arm64 f1tenth-sensor-policy:sensor-policy
    retag f1tenth-sensor-policy:gigaflow-folded-sophy-arm64 f1tenth-sensor-policy:gigaflow-folded-sophy
    retag f1tenth-sensor-policy:sophy-courtyard-65a-arm64 f1tenth-sensor-policy:sophy-courtyard-65a
    retag f1tenth-racing:develop-arm64 f1tenth-racing:develop
    echo "==> torch pins (no Jetson GPU on this print if runtime nvidia is absent)"
    for img in f1tenth-gigaflow:gigaflow f1tenth-sensor-policy:sensor-policy f1tenth-racing:develop; do
      if docker image inspect "\${img}" >/dev/null 2>&1; then
        echo "--- \${img} ---"
        docker run --rm --entrypoint python3 "\${img}" -c \
          'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())' || true
      fi
    done
    ;;
  *)
    if [ -f "\${IMAGES}/f1tenth-gigaflow-all-models.tar.gz" ]; then
      load_tar "\${IMAGES}/f1tenth-gigaflow-all-models.tar.gz"
    else
      load_tar "\${IMAGES}/f1tenth-gigaflow.tar.gz"
    fi
    if [ -f "\${IMAGES}/f1tenth-sensor-policy-all-models.tar.gz" ]; then
      load_tar "\${IMAGES}/f1tenth-sensor-policy-all-models.tar.gz"
    else
      load_tar "\${IMAGES}/f1tenth-sensor-policy.tar.gz"
    fi
    if [ "\${SKIP_TRAINER}" != "1" ]; then
      load_tar "\${IMAGES}/f1tenth-courtyard-trainer.tar.gz"
    fi
    load_tar "\${IMAGES}/f1tenth-racing.tar.gz"
    ;;
esac
docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' \
  | grep -E 'f1tenth-(gigaflow|sensor-policy|courtyard-trainer|racing)' || true
EOS
)"

echo "==> 4. write /opt/f1tenth/run wrappers"
remote "$(cat <<'EOS'
set -euo pipefail
RUN=/opt/f1tenth/run
mkdir -p "${RUN}"

write_gigaflow() {
  local name="$1"
  local image="$2"
  local ckpt="$3"
  cat > "${RUN}/${name}" <<RUN
#!/usr/bin/env bash
set -euo pipefail
IMAGE="${image}"
CKPT="${ckpt}"
DEVICE="\${DEVICE:-cuda}"
if fuser /dev/sensors/vesc >/dev/null 2>&1; then
  echo "VESC serial device already has an owner; refusing to start." >&2
  fuser -v /dev/sensors/vesc >&2 || true
  exit 1
fi
exec docker run --rm -it --runtime nvidia --privileged --net=host \\
  -e CHECKPOINT="\${CKPT}" \\
  -v /dev:/dev \\
  -v /opt/f1tenth/config:/config:ro \\
  -v /opt/f1tenth/policies:/policies:ro \\
  "\${IMAGE}" \\
  ros2 launch f1tenth_bringup gigaflow_policy.launch.py \\
    checkpoint_path:=\${CKPT} device:=\${DEVICE}
RUN
  chmod +x "${RUN}/${name}"
}

write_sophy() {
  local name="$1"
  local image="$2"
  local ckpt="$3"
  cat > "${RUN}/${name}" <<RUN
#!/usr/bin/env bash
set -euo pipefail
IMAGE="${image}"
CKPT="${ckpt}"
DEVICE="\${DEVICE:-cuda}"
if fuser /dev/sensors/vesc >/dev/null 2>&1; then
  echo "VESC serial device already has an owner; refusing to start." >&2
  fuser -v /dev/sensors/vesc >&2 || true
  exit 1
fi
exec docker run --rm -it --runtime nvidia --privileged --net=host \\
  -e CHECKPOINT="\${CKPT}" \\
  -v /dev:/dev \\
  -v /opt/f1tenth/config:/config:ro \\
  -v /opt/f1tenth/policies:/policies:ro \\
  "\${IMAGE}" \\
  ros2 launch f1tenth_bringup sensor_policy.launch.py \\
    checkpoint_path:=\${CKPT} device:=\${DEVICE}
RUN
  chmod +x "${RUN}/${name}"
}

write_gigaflow run_courtyard_sleepy_7lap.sh \
  f1tenth-gigaflow:courtyard-sleepy-7lap \
  /policies/courtyard_sleepy_s2_dagger_7lap.pt
write_gigaflow run_courtyard_pr104_6lap.sh \
  f1tenth-gigaflow:courtyard-pr104-6lap \
  /policies/courtyard_pr104_hybrid_bc_6lap.pt
write_gigaflow run_irl_awac_kl.sh \
  f1tenth-gigaflow:irl-awac-kl \
  /policies/irl_awac_kl_kl2x_winner.pt
write_gigaflow run_gigaflow_base_5700.sh \
  f1tenth-gigaflow:base-5700 \
  /policies/actor_005700_irl_baked.pt
write_sophy run_sophy_folded_gigaflow.sh \
  f1tenth-sensor-policy:gigaflow-folded-sophy \
  /policies/gigaflow_folded_sophy_8m.pt
write_sophy run_sophy_courtyard_65a.sh \
  f1tenth-sensor-policy:sophy-courtyard-65a \
  /policies/policy_875520000.pt

cat > "${RUN}/run_courtyard_online_trainer.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
IMAGE="${IMAGE:-f1tenth-courtyard-trainer:courtyard-trainer}"
CKPT="${CHECKPOINT:-/opt/f1tenth/policies/actor_005700_irl_baked.pt}"
OUT="${RUN_DIR:-/opt/f1tenth/outputs/courtyard_ppo}"
MODE="${COURTYARD_MODE:-finetune}"
mkdir -p "$(dirname "${CKPT}")" "${OUT}"
if [ ! -f "${CKPT}" ]; then
  echo "copy actor checkpoint to ${CKPT}" >&2
  exit 1
fi
if [ "${MODE}" = "ppo" ] || [ "${MODE}" = "train" ]; then
  EXTRA=( )
else
  EXTRA=( )
fi
exec docker run --rm --gpus all \
  -e CHECKPOINT="${CKPT}" \
  -e RUN_DIR="${OUT}" \
  -e COURTYARD_MODE="${MODE}" \
  -v /opt/f1tenth/policies:/opt/f1tenth/policies:ro \
  -v /opt/f1tenth/outputs/courtyard_ppo:/opt/f1tenth/outputs/courtyard_ppo \
  "${IMAGE}" "$@"
RUN
chmod +x "${RUN}/run_courtyard_online_trainer.sh"

cat > "${RUN}/run_racing_392d.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
IMAGE="${IMAGE:-f1tenth-racing:develop}"
if fuser /dev/sensors/vesc >/dev/null 2>&1; then
  echo "VESC serial device already has an owner; refusing to start." >&2
  fuser -v /dev/sensors/vesc >&2 || true
  exit 1
fi
exec docker run --rm -it --runtime nvidia --privileged --net=host \
  -v /dev:/dev \
  -v /opt/f1tenth/config:/config:ro \
  -v /opt/f1tenth/policies:/policies:ro \
  "${IMAGE}" \
  ros2 launch f1tenth_bringup race.launch.py
RUN
chmod +x "${RUN}/run_racing_392d.sh"

ls -la "${RUN}"
EOS
)"

cat <<EOF

==> ${DRY_RUN:+DRY-RUN }install plan for ${HOST} complete
    remote bundle: ${REMOTE_BUNDLE}
    policies:      ${REMOTE_POLICIES}
    params:        ${REMOTE_CONFIG}/params.yaml
    wrappers:      ${REMOTE_RUN}

Once Arjun gives the go (NOT before — do not SSH until then):
  ${BUNDLE_DIR}/install_to_shereef_f1tenth.sh ${HOST} ${CAR}

Then on the car, pick a model:
  ssh ${HOST} ${REMOTE_RUN}/run_courtyard_sleepy_7lap.sh
  ssh ${HOST} ${REMOTE_RUN}/run_courtyard_pr104_6lap.sh
  ssh ${HOST} ${REMOTE_RUN}/run_irl_awac_kl.sh
  ssh ${HOST} ${REMOTE_RUN}/run_gigaflow_base_5700.sh
  ssh ${HOST} ${REMOTE_RUN}/run_sophy_folded_gigaflow.sh
  ssh ${HOST} ${REMOTE_RUN}/run_sophy_courtyard_65a.sh
  ssh ${HOST} ${REMOTE_RUN}/run_courtyard_online_trainer.sh
  ssh ${HOST} ${REMOTE_RUN}/run_racing_392d.sh
EOF
